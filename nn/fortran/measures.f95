module measures
    use config
    use propagate
    use util
    use init

    implicit none

    real(8),external :: dnrm2

    contains

        real(8) function loss(flat_weights,nw,set_type)
            
            implicit none

            integer,intent(in) :: nw,set_type
            real(8),intent(in) :: flat_weights(1:nw)

            !* scratch
            integer :: conf,atm
            real(8) :: tmp_energy,tmp_forces,tmp_reglrn

            !* read in NN weights
            call parse_array_to_structure(flat_weights,net_weights)

            do conf=1,data_sets(set_type)%nconf,1
                do atm=1,data_sets(set_type)%configs(conf)%n,1
                    !* forward prop on training data
                    call forward_propagate(conf,atm,set_type)
                end do
            end do

            tmp_energy = loss_energy(set_type)
            tmp_forces = loss_forces(set_type)
            tmp_reglrn = loss_reglrn(flat_weights)

            loss = tmp_energy + tmp_forces + tmp_reglrn
        end function loss

        subroutine loss_jacobian(flat_weights,nw,set_type,jacobian)

            implicit none
            
            integer,intent(in) :: nw,set_type
            real(8),intent(in) :: flat_weights(1:nw)
            real(8),intent(out) :: jacobian(1:nw)

            !* scratch
            integer :: conf,atm
            real(8) :: tmpE

            type(weights) :: loss_jac
            type(weights) :: tmp_jac

            call allocate_weights(loss_jac)
            call allocate_weights(tmp_jac)
        
            call zero_weights(loss_jac)

            !* read in supplied weights
            call parse_array_to_structure(flat_weights,net_weights)

            do conf=1,data_sets(set_type)%nconf,1
                
                call zero_weights(tmp_jac)
                
                do atm=1,data_sets(set_type)%configs(conf)%n,1
                    !* forward prop on training data
                    call forward_propagate(conf,atm,set_type)
                    
                    call backward_propagate(conf,atm,set_type)
                   
                    !* total energy contribution
                    call loss_energy_jacobian(tmp_jac)
                end do

                !-----------------------!
                !* energy contribution *!
                !-----------------------!

                !* sgn( \sum_i E_i - E ) * \sum_i dE_i / dw
                tmpE = sum(data_sets(set_type)%configs(conf)%current_ei) &
                        &-data_sets(set_type)%configs(conf)%energy
               
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
            end do

            !-------------------------------!
            !* regularization contribution *!
            !-------------------------------!
        
            call loss_reglrn_jacobian(loss_jac)

            !* structured to 1d
            call parse_structure_to_array(loss_jac,jacobian)

            call deallocate_weights(loss_jac)
            call deallocate_weights(tmp_jac)
        end subroutine loss_jacobian

        real(8) function loss_energy(set_type)
            implicit none
            
            integer,intent(in) :: set_type

            !* scratch
            real(8) :: tmp1,tmp2,tot_energy_loss
            integer :: conf

            tot_energy_loss = 0.0d0

            do conf=1,data_sets(set_type)%nconf,1
    
                !* 1/Natm
                tmp2 = 1.0d0/dble(data_sets(set_type)%configs(conf)%n)

                !* consider energy per atom
                tmp1 = abs(sum(data_sets(set_type)%configs(conf)%current_ei)&
                        & - data_sets(set_type)%configs(conf)%energy) * tmp2
                
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

            tot_forces_loss = 0.0d0

            !* could fill 1-d arrays and use lapack here

            do conf=1,data_sets(set_type)%nconf,1
                do atm=1,data_sets(set_type)%configs(conf)%n,1
                    do ii=1,3,1
                        tmp = abs(data_sets(set_type)%configs(conf)%current_fi(ii,atm) - &
                        & data_sets(set_type)%configs(conf)%forces(ii,atm))

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

            !* l2 norm**2 = w.T w
            loss_reglrn = dnrm2(size(flat_weights),flat_weights,1)**2 * 0.5d0 * &
                    &loss_const_reglrn
        end function loss_reglrn


        subroutine loss_reglrn_jacobian(loss_jac)
            implicit none

            type(weights),intent(inout) :: loss_jac

            !* scratch
            integer :: ii,jj
            
            !* layer 1
            do ii=1,net_dim%hl1
                do jj=1,D+1
                    loss_jac%hl1(jj,ii) = loss_jac%hl1(jj,ii) + net_weights%hl1(jj,ii)*&
                            &loss_const_reglrn
                            !&dydw%hl1(jj,ii)*loss_const_reglrn
                end do
            end do
            
            !* layer 2
            do ii=1,net_dim%hl2
                do jj=1,net_dim%hl1+1
                    loss_jac%hl2(jj,ii) = loss_jac%hl2(jj,ii) + net_weights%hl2(jj,ii)*&
                            &loss_const_reglrn
                            !&dydw%hl2(jj,ii)*loss_const_reglrn
                end do
            end do

            !* final layer
            do ii=1,net_dim%hl2+1,1
                !loss_jac%hl3(ii) = loss_jac%hl3(ii) + net_weights%hl3(ii)*dydw%hl3(ii)*&
                loss_jac%hl3(ii) = loss_jac%hl3(ii) + net_weights%hl3(ii)*&
                            &loss_const_reglrn
            end do
        end subroutine loss_reglrn_jacobian

        subroutine loss_energy_jacobian(tmp_jac)
            implicit none

            type(weights),intent(inout) :: tmp_jac

            tmp_jac%hl1 = tmp_jac%hl1 + dydw%hl1
            tmp_jac%hl2 = tmp_jac%hl2 + dydw%hl2
            tmp_jac%hl3 = tmp_jac%hl3 + dydw%hl3
            
        end subroutine loss_energy_jacobian
end module measures