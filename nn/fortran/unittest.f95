program unittest

    use config
    use init
    use propagate
    use util
    use io

    implicit none

    call main()

    contains
        subroutine main()
            implicit none

            logical :: tests(1:1)
            
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

            num_tests = 1

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

            deallocate(features)
            deallocate(forces)
            deallocate(energy)

            do ii=1,num_tests
                call unittest_test(ii,tests(ii))    
            end do

            call unittest_summary(tests)
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
            call parse_weights_flatten(original_weights,nwght,.false.)
            
            conf = 1
            set_type = 1
            atm = 1

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
                        call parse_weights_expand(original_weights,nwght)

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
               
                !* analytical derivatives

                call parse_weights_expand(original_weights,nwght)

                !* forward prop
                call forward_propagate(conf,atm,set_type)
                
                !* back prop
                call backward_propagate(conf,atm,set_type)
              
                !* parse dy/dw into 1d array
                call parse_weights_flatten(dydw_flat,nwght,.true.)

                !do ii=1,nwght,1
                !    write(*,*) dw,ii,derivs(ii),dydw_flat(ii)
                !end do
                test_result = array_equal(derivs,dydw_flat,dble(1e-15),dble(1e-10))
            end do
            test_dydw = test_result
        end function

end program unittest
