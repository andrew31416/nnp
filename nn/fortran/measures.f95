module measures
    use config
    use propagate
    use util
    use init

    implicit none

    real(8),external :: dnrm2

    contains

        real(8) function loss(flat_weights,set_type)
            
            implicit none

            integer,intent(in) :: set_type
            real(8),intent(in) :: flat_weights(:)

            !* scratch
            integer :: conf
            real(8) :: tmp_energy,tmp_forces,tmp_reglrn
            
            !* read in NN weights
            call parse_array_to_structure(flat_weights,net_weights)
            call copy_weights_to_nobiasT()

            do conf=1,data_sets(set_type)%nconf,1
                if (allocated(dydx)) then
                    deallocate(dydx)
                end if
                call allocate_dydx(set_type,conf)
                call allocate_units(set_type,conf)
               
                call forward_propagate(set_type,conf)
                
                call backward_propagate(set_type,conf)                
                !do atm=1,data_sets(set_type)%configs(conf)%n,1
                !    !* forward prop on data
                !    call forward_propagate(conf,atm,set_type)
                !    
                !    !* backward prop on data
                !    call backward_propagate(conf,atm,set_type)
                !end do

                !* calculate forces in configuration
                call calculate_forces(set_type,conf)
            end do

            !* calculate predicted forces
            !call backprop_all_forces(set_type)

            tmp_energy = loss_energy(set_type)
            tmp_forces = loss_forces(set_type)
            tmp_reglrn = loss_reglrn(flat_weights)
            
            loss = tmp_energy + tmp_forces + tmp_reglrn
        end function loss

        subroutine loss_jacobian(flat_weights,set_type,jacobian)

            implicit none
            
            integer,intent(in) :: set_type
            real(8),intent(in) :: flat_weights(:)
            real(8),intent(out) :: jacobian(:)

            !* scratch
            integer :: conf,atm
            real(8) :: tmpE
            logical :: include_force_loss

            type(weights) :: loss_jac
            type(weights) :: tmp_jac
            type(weights) :: tmp2_jac
            
            call allocate_weights(loss_jac)
            call allocate_weights(tmp_jac)
            call allocate_weights(tmp2_jac)
        
            call zero_weights(loss_jac)

            !* read in supplied weights
            call parse_array_to_structure(flat_weights,net_weights)
            call copy_weights_to_nobiasT()
            !* initialise force loss subsidiary mem.
            call init_forceloss_subsidiary_mem()

            !* decide whether or not to compute force loss derivatives
            if (scalar_equal(loss_const_forces,0.0d0,dble(1e-15),dble(1e-10)**2,.false.)) then
                include_force_loss = .false.
            else
                include_force_loss = .true.
            end if
            
            call allocate_weights(d2ydxdw)

            do conf=1,data_sets(set_type)%nconf,1
                if(allocated(dydx)) then
                    deallocate(dydx)
                end if
                call allocate_dydx(set_type,conf)
                call zero_weights(tmp_jac)
                call zero_weights(tmp2_jac)
                !call allocate_d2ydxdw_mem(conf,set_type,d2ydxdw)
                
                
                call forward_propagate(set_type,conf)
                call backward_propagate(set_type,conf)
                
                if (include_force_loss) then
                    call calculate_forces(set_type,conf)
                end if

                do atm=1,data_sets(set_type)%configs(conf)%n,1
                    !* for energy contribuion to loss
                    call calculate_dydw(set_type,conf,atm)
                   
                    !* total energy contribution
                    call loss_energy_jacobian(tmp_jac)
                end do
                

                !-----------------------!
                !* energy contribution *!
                !-----------------------!

                !* sgn( \sum_i E_i - E ) * \sum_i dE_i / dw
                tmpE = sum(data_sets(set_type)%configs(conf)%current_ei) &
                        &-data_sets(set_type)%configs(conf)%ref_energy
               
                if(loss_norm_type.eq.1) then
                    !* l1 norm
                    tmpE = sign(1.0d0,tmpE) / dble(data_sets(set_type)%configs(conf)%n)
                else if (loss_norm_type.eq.2) then
                    !* l2 norm
                    tmpE = sign(1.0d0,tmpE)*tmpE / dble(data_sets(set_type)%configs(conf)%n)
                end if

                !* constant scaling
                tmpE = tmpE * 0.5d0 * loss_const_energy
                
                loss_jac%hl1 = loss_jac%hl1 + tmp_jac%hl1 * tmpE
                loss_jac%hl2 = loss_jac%hl2 + tmp_jac%hl2 * tmpE
                loss_jac%hl3 = loss_jac%hl3 + tmp_jac%hl3 * tmpE

            

                if (include_force_loss) then
                    !=======================!
                    !* forces contribution *!
                    !=======================!
                    call loss_forces_jacobian(set_type,conf,tmp2_jac)
                    
                    loss_jac%hl1 = loss_jac%hl1 + tmp2_jac%hl1 * loss_const_forces * 0.5d0 
                    loss_jac%hl2 = loss_jac%hl2 + tmp2_jac%hl2 * loss_const_forces * 0.5d0
                    loss_jac%hl3 = loss_jac%hl3 + tmp2_jac%hl3 * loss_const_forces * 0.5d0
                end if

                deallocate(dydx)
            end do !* end loop over confs
            
            !-------------------------------!
            !* regularization contribution *!
            !-------------------------------!
        
            call loss_reglrn_jacobian(loss_jac)

            !* structured to 1d
            call parse_structure_to_array(loss_jac,jacobian)

            call deallocate_weights(loss_jac)
            call deallocate_weights(tmp_jac)
            call deallocate_weights(tmp2_jac)
            call deallocate_forceloss_subsidiary_mem()                
        end subroutine loss_jacobian

        real(8) function loss_energy(set_type)
            implicit none
            
            integer,intent(in) :: set_type

            !* scratch
            real(8) :: tmp1,tmp2,tot_energy_loss
            integer :: conf

            if (scalar_equal(loss_const_energy,0.0d0,dble(1e-18),dble(1e-18),.false.)) then
                loss_energy = 0.0d0
                return
            end if

            tot_energy_loss = 0.0d0
            
            do conf=1,data_sets(set_type)%nconf,1
    
                !* 1/Natm
                tmp2 = 1.0d0/dble(data_sets(set_type)%configs(conf)%n)

                !* consider energy per atom
                tmp1 = abs(sum(data_sets(set_type)%configs(conf)%current_ei)&
                        & - data_sets(set_type)%configs(conf)%ref_energy) * tmp2
                
                if (loss_norm_type.eq.1) then
                    !* l1 norm
                    tot_energy_loss = tot_energy_loss + tmp1
                else if (loss_norm_type.eq.2) then
                    !* l2 norm
                    tot_energy_loss = tot_energy_loss + tmp1**2
                end if

            end do
            
            loss_energy = tot_energy_loss * 0.5d0 * loss_const_energy 
        end function loss_energy

        real(8) function loss_forces(set_type)
            implicit none

            integer,intent(in) :: set_type

            !* scratch
            integer :: conf,atm,ii
            real(8) :: tmp,tot_forces_loss
            
            if (scalar_equal(loss_const_forces,0.0d0,dble(1e-18),dble(1e-18),.false.)) then
                loss_forces = 0.0d0
                return
            end if
            
            tot_forces_loss = 0.0d0

            !* could fill 1-d arrays and use lapack here

            do conf=1,data_sets(set_type)%nconf,1
                do atm=1,data_sets(set_type)%configs(conf)%n,1
                    do ii=1,3,1
                        tmp = abs(data_sets(set_type)%configs(conf)%current_fi(ii,atm) - &
                        & data_sets(set_type)%configs(conf)%ref_fi(ii,atm))
                        
                        if (loss_norm_type.eq.1) then
                            tot_forces_loss = tot_forces_loss + tmp
                        else
                            tot_forces_loss = tot_forces_loss + tmp**2
                        end if
                    end do
                end do
            end do
            loss_forces = tot_forces_loss * 0.5d0 * loss_const_forces
        end function loss_forces

        real(8) function loss_reglrn(flat_weights)
            implicit none

            real(8),intent(in) :: flat_weights(:)
            
            if (scalar_equal(loss_const_reglrn,0.0d0,dble(1e-18),dble(1e-18),.false.)) then
                loss_reglrn = 0.0d0
                return
            end if

            !* l2 norm**2 = w.T w
            loss_reglrn = dnrm2(size(flat_weights),flat_weights,1)**2 * 0.5d0 * &
                    &loss_const_reglrn
        end function loss_reglrn


        subroutine loss_reglrn_jacobian(loss_jac)
            implicit none

            type(weights),intent(inout) :: loss_jac

            !* scratch
            integer :: ii,jj

            if (scalar_equal(loss_const_reglrn,0.0d0,dble(1e-18),dble(1e-18),.false.)) then
                return
            end if
            
            !* layer 1
            do ii=0,D
                do jj=1,net_dim%hl1
                    loss_jac%hl1(jj,ii) = loss_jac%hl1(jj,ii) + net_weights%hl1(jj,ii)*&
                            &loss_const_reglrn
                end do
            end do
            
            !* layer 2
            do ii=0,net_dim%hl1
                do jj=1,net_dim%hl2
                    loss_jac%hl2(jj,ii) = loss_jac%hl2(jj,ii) + net_weights%hl2(jj,ii)*&
                            &loss_const_reglrn
                end do
            end do

            !* final layer
            do ii=0,net_dim%hl2
                loss_jac%hl3(ii) = loss_jac%hl3(ii) + net_weights%hl3(ii)*&
                            &loss_const_reglrn
            end do
        end subroutine loss_reglrn_jacobian

        subroutine loss_energy_jacobian(tmp_jac)
            implicit none

            type(weights),intent(inout) :: tmp_jac
            
            if (scalar_equal(loss_const_energy,0.0d0,dble(1e-18),dble(1e-18),.false.)) then
                return
            end if

            tmp_jac%hl1 = tmp_jac%hl1 + dydw%hl1
            tmp_jac%hl2 = tmp_jac%hl2 + dydw%hl2
            tmp_jac%hl3 = tmp_jac%hl3 + dydw%hl3
        end subroutine loss_energy_jacobian

        subroutine loss_forces_jacobian(set_type,conf,tmp_jac)
            implicit none

            integer,intent(in) :: set_type,conf
            type(weights),intent(inout) :: tmp_jac
            
            integer :: atm,kk,num_neigh,ii,idx,dd
            real(8) :: vec(1:3),sgns(1:3,1:data_sets(set_type)%configs(conf)%n)
            real(8) :: tmp_buffer(1:3)

            if (scalar_equal(loss_const_forces,0.0d0,dble(1e-18),dble(1e-18),.false.)) then
                return
            end if
            do atm=1,data_sets(set_type)%configs(conf)%n
                do dd=1,3
                    sgns(dd,atm) = sign(1.0d0,data_sets(set_type)%configs(conf)%current_fi(dd,atm)-&
                            &data_sets(set_type)%configs(conf)%ref_fi(dd,atm))
                end do
            end do

            do atm=1,data_sets(set_type)%configs(conf)%n
                call forceloss_weight_derivative_subsidiary1(atm)
                
                do kk=1,D
                    num_neigh = data_sets(set_type)%configs(conf)%x_deriv(kk,atm)%n 

                    if (num_neigh.le.0) then
                        !* no atoms contribute to this feature
                        cycle
                    end if
                    
                    call forceloss_weight_derivative_subsidiary2(set_type,conf,atm,kk)

                    tmp_buffer = 0.0d0

                    do ii=1,num_neigh
                        !* identity of neighbouring atom
                        idx = data_sets(set_type)%configs(conf)%x_deriv(kk,atm)%idx(ii)
                        
                        !* d feature / d r_idx
                        vec = data_sets(set_type)%configs(conf)%x_deriv(kk,atm)%vec(:,ii)
                       
                        do dd=1,3 
                            tmp_buffer(dd) = tmp_buffer(dd) + vec(dd)*sgns(dd,idx)
                        end do 
                        !do dd=1,3
                        !    sgn(dd) = sign(1.0d0,data_sets(set_type)%configs(conf)%current_fi(dd,idx)-&
                        !            &data_sets(set_type)%configs(conf)%ref_fi(dd,idx))
                        !end do

    
                        !do dd=1,3
                        !    tmp_jac%hl1 = tmp_jac%hl1 - vec(dd)*sgns(dd,idx)*d2ydxdw(atm,kk)%hl1
                        !    tmp_jac%hl2 = tmp_jac%hl2 - vec(dd)*sgns(dd,idx)*d2ydxdw(atm,kk)%hl2
                        !    tmp_jac%hl3 = tmp_jac%hl3 - vec(dd)*sgns(dd,idx)*d2ydxdw(atm,kk)%hl3
                        !end do
                    end do !* end loop over neighbours to (kk,atm_loc)
                    
                    do dd=1,3
                        tmp_jac%hl1 = tmp_jac%hl1 - tmp_buffer(dd)*d2ydxdw%hl1
                        tmp_jac%hl2 = tmp_jac%hl2 - tmp_buffer(dd)*d2ydxdw%hl2
                        tmp_jac%hl3 = tmp_jac%hl3 - tmp_buffer(dd)*d2ydxdw%hl3
                    end do

                end do !* end loop over features           
            end do  !* end loop over atoms in local cell
        end subroutine loss_forces_jacobian
end module measures
