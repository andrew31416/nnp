program unittest

    use config
    use init
    use propagate
    use util
    use io
    use measures

    implicit none

    call main()

    contains
        subroutine main()
            implicit none

            logical :: tests(1:4)
            
            !* net params
            integer :: num_nodes(1:2),nlf_type,fD

            !* features
            real(8),allocatable :: features(:,:,:)
            real(8),allocatable :: forces(:,:,:)
            real(8),allocatable :: energy(:,:)
            integer,allocatable :: slice_idx(:,:)
            integer :: natm,nconf

            !* scratch
            integer :: ii

            integer :: num_tests

            num_tests = 4

            !* number of nodes
            num_nodes(1) = 3
            num_nodes(2) = 2

            !* nonlinear function
            nlf_type = 1
            
            !* features
            fD = 2
            natm = 2
            nconf = 2
       
            call unittest_header()

            allocate(features(fD,natm*nconf,2))
            allocate(forces(3,natm*nconf,2))
            allocate(energy(nconf,2))
            allocate(slice_idx(2,nconf))
            call srand(1)
            call random_number(features(:,:,:))
            call random_number(forces(:,:,:))
            call random_number(energy(:,:))

            call initialise_net(num_nodes,nlf_type,fD)

            do ii=1,nconf,1
                slice_idx(1,ii) = (ii-1)*natm + 1
                slice_idx(2,ii) = ii*natm
            end do

            do ii=1,2,1
                call initialise_set(ii,nconf,nconf*natm,slice_idx,features(:,:,ii),&
                        &forces(:,:,ii),energy(:,ii))
            end do

            !----------------------!
            !* perform unit tests *!
            !----------------------!
            
            tests(1) = test_dydw()
            
            !* tests(2-4)
            call test_loss_jac(tests(2:4))

            deallocate(features)
            deallocate(forces)
            deallocate(energy)

            do ii=1,num_tests
                call unittest_test(ii,tests(ii))    
            end do

            call unittest_summary(tests)

            if (all(tests)) then
                !* success
                call exit(1)
            else
                !* failure
                call exit(0)
            end if
        end subroutine main

        logical function test_dydw()
            implicit none

            logical :: test_result
    
            !* scratch
            integer :: ww,ii,jj,atm,conf,set_type
            real(8),allocatable :: original_weights(:)
            real(8),allocatable :: dydw_flat(:)
            real(8),allocatable :: derivs(:)
            real(8) :: dw,w0,dy,num_val

            allocate(original_weights(nwght))
            allocate(dydw_flat(nwght))
            allocate(derivs(nwght))

            !* initial weights
            call parse_structure_to_array(net_weights,original_weights)
            
            conf = 1
            set_type = 1
            atm = 1

            dy = 0.0d0

            do ww=4,4,1
                dw = 1.0d0/(10.0d0**ww)

                do ii=1,nwght,1
                    w0 = original_weights(ii)
                    do jj=1,2,1
                        if (jj.eq.1) then
                            original_weights(ii) = w0 + dw
                        else 
                            original_weights(ii) = w0 - dw
                        end if
                        
                        !* read in weights
                        call parse_array_to_structure(original_weights,net_weights)
                        
                        !* forward propagate
                        call forward_propagate(conf,atm,set_type)

                        if (jj.eq.1) then
                            dy = data_sets(set_type)%configs(conf)%current_ei(atm)
                        else 
                            dy = dy - data_sets(set_type)%configs(conf)%current_ei(atm)
                        end if
                    end do

                    num_val = dy/(2.0d0*dw)
 
                    derivs(ii) = num_val
               
                    !* return to initial value
                    original_weights(ii) = w0
                end do
               
                !--------------------------!
                !* analytical derivatives *!
                !--------------------------!

                call parse_array_to_structure(original_weights,net_weights)

                !* forward prop
                call forward_propagate(conf,atm,set_type)
                
                !* back prop
                call backward_propagate(conf,atm,set_type)
              
                !* parse dy/dw into 1d array
                call parse_structure_to_array(dydw,dydw_flat)

                test_result = array_equal(derivs,dydw_flat,dble(1e-15),dble(1e-10))
            end do
            test_dydw = test_result
        end function test_dydw

        subroutine test_loss_jac(test_result)
            !* test the jacobian of energy, force, reg. loss functions *!

            implicit none

            logical,intent(out) :: test_result(1:3)

            !* scratch
            integer :: ii,jj,kk,ww,conf,atm,set_type
            real(8) :: dw,w0,dloss,tmp
            real(8),dimension(:),allocatable :: num_jac,anl_jac,original_weights

            allocate(num_jac(nwght))
            allocate(anl_jac(nwght))
            allocate(original_weights(nwght))
            
            !* initial weights
            call parse_structure_to_array(net_weights,original_weights)

            conf= 1
            atm = 1
            set_type = 1

            dloss = 0.0d0

            do ii=1,3,1
                if (ii.eq.1) then
                    loss_const_energy = 1.0d0
                    loss_const_forces = 0.0d0
                    loss_const_reglrn = 0.0d0
                else if (ii.eq.2) then
                    loss_const_energy = 0.0d0
                    loss_const_forces = 1.0d0
                    loss_const_reglrn = 0.0d0
                else
                    loss_const_energy = 0.0d0
                    loss_const_forces = 0.0d0
                    loss_const_reglrn = 1.0d0
                end if
                
                do ww=4,4,1
                    !* finite difference
                    dw = 1.0d0/(10.0d0**(ww))

                    do jj=1,nwght,1
                        w0 = original_weights(jj)

                        do kk=1,2,1
                            if (kk.eq.1) then
                                original_weights(jj) = w0 + dw
                            else
                                original_weights(jj) = w0 - dw
                            end if
                            
                            tmp = loss(original_weights,nwght,set_type)

                            if (kk.eq.1) then
                                dloss = tmp
                            else
                                dloss = dloss - tmp
                            end if

                            original_weights(jj) = w0
                        end do !* end loop +/- dw

                        num_jac(jj) = dloss / (2.0d0 * dw)
                    end do !* end loop over weights

                end do !* end loop over dw

                !-------------------------------!
                !* analytical jacobian of loss *!
                !-------------------------------!

                call loss_jacobian(original_weights,nwght,set_type,anl_jac)
                do ww=1,5
                    write(*,*) ww,num_jac(ww),anl_jac(ww)
                end do
                test_result(ii) = array_equal(num_jac,anl_jac,dble(1e-15),dble(1e-10))
            end do !* end loop over loss terms
        end subroutine test_loss_jac

end program unittest
